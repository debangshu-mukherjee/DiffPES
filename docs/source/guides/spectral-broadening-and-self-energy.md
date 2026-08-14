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

```{figure} figures/broadening-profiles.png
:alt: Gaussian, Voigt, and near-Lorentzian energy profiles
:width: 74%

The normalized profiles. The Gaussian carries the instrument resolution,
the Lorentzian carries lifetime broadening, and the Voigt is their true
convolution.
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

```{figure} figures/broadening-linewidth-ek.png
:alt: Graphene spectra with 30 meV and 250 meV constant self-energy

The same graphene bands under a 30 meV and a 250 meV constant
self-energy. The linewidth is the imaginary part of $\Sigma$ made visible.
```

## Instrument Transmission and Resolution

`apply_transmission` evaluates the calibrated analyser response at true
kinetic energy. Its monotone log-polynomial shape is normalized over the fixed
`DetectorCalibration.transmission_reference_domain_ev`, so caller crops and
padding cannot alter retained values. Apply it before detector resolution.

`apply_resolution` then convolves detector-bin densities in native `(u, v, E)`
coordinates. It integrates the continuous Gaussian over explicit source and
target bin edges. The calibration owns the FWHM values. The result includes
the blurred density and captured-flux fractions under the `loss` policy:

```python
transmitted = diffpes.simul.apply_transmission(
    detector_density,
    kinetic_energy_axis_ev,
    raw_transmission_slopes,
    calibration,
)
blurred, captured_fractions, valid = diffpes.simul.apply_resolution(
    transmitted,
    calibration,
)
```

```{figure} figures/broadening-resolution.png
:alt: Intrinsic spectrum versus the same spectrum with Gaussian energy resolution

An intrinsic spectrum before and after a 120 meV Gaussian energy
resolution through `convolve_energy`. Lifetime and resolution broadening
are separate operations with different lineshapes.
```

`convolve_energy` and `convolve_momentum_map` are sampled, uniform-grid
SciPy/Chinook parity approximations. The latter requires explicit Cartesian
momentum axes in inverse angstroms; it never interprets fractional `KGrid`
coordinates as physical spacings. `convolve_kpath` is the physical-k cut
operator for nonuniform path cells. It uses the same analytic finite-volume
semantics as native resolution and reports escaped boundary mass rather than
renormalizing rows. A calibrated angular width should always use
`apply_resolution` after the detector map.

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
In `bulk_kz` mode, the wrapped out-of-plane-momentum integral reduces these
occupied intrinsic intensities before transmission and native detector
resolution. It is mutually exclusive with the `coherent_slab` depth-amplitude
sum; see [kz Broadening and Photon-Energy
Scans](kz-broadening-and-photon-energy-scans.md).

`band_group_weight_sensitivity` returns derivatives of the matrix-element
weights before spectral, exposure, background, or detector factors.
`map_source_to_detector` converts a self-describing source carrier into an
explicit `DetectorCalibration` target and reports captured boundary flux.
`apply_detector_effects` performs the complete deterministic detector chain,
while `simulate_arpes` and `simulate_arpes_cut` join that chain to the coherent
resolvent source assembly. `expected_counts` remains the narrower
post-resolution density-to-count primitive.

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

See [Simulating ARPES Spectra](simulating-arpes-spectra.md) for the complete
driver pipeline and
[JAX Transformability and Gradients](jax-transformability-and-gradients.md)
for derivative semantics.
