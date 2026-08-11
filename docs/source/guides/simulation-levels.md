# Coherent Spectral Assembly

diffpes preserves complex transition amplitudes until the physically required
outgoing-channel reduction. A VASP `PROCAR` projection contains only
probabilities $|c_{no}(\mathbf{k})|^2$ and cannot reconstruct the relative
orbital or atomic-centre phases needed for interference. The former
projection-probability spectrum tiers and level-string dispatcher are
therefore removed rather than presented as quantitative ARPES workflows.

## Matrix Elements Before Spectral Reduction

The coherent path uses functions from `diffpes.simul.matrixel`:

1. Build an `OrbitalBasis`, `RadialSpec`, `MatrixElementParams`,
   `RadialQuadratureSpec`, and `FinalStateSpec`.
2. Supply explicit vacuum final momenta and their emission-validity mask.
3. Use `assemble_orbital_transition_channels` to retain one complex Cartesian
   transition vector per orbital and outgoing spin.
4. Contract polarization only after assembling the transition vector.
5. Use `project_band_channels` with complex band coefficients.
6. Apply `matrix_element_intensity` exactly once when a band-weight fast path
   is valid, or retain full transition sources for the resolvent.

This ordering preserves orbital, sublattice, atomic-centre, polarization, and
surface-attenuation interference.

## Two Coherent Spectral Paths

`spectral_intensity_resolvent` evaluates

$$
-\frac{1}{\pi}\operatorname{Im}
\sum_\alpha s_\alpha^\dagger
[(\omega+i\eta-\Sigma)I-H]^{-1}s_\alpha ,
$$

without differentiating eigenvectors. It is the certified path at exact or
near degeneracy.

`spectral_intensity_eigen` consumes eigenvalues and gauge-invariant band
weights. It is faster away from degeneracies, where those weights have
already been formed from the coherent matrix-element channels.

Both paths consume the retarded self-energy from `evaluate_self_energy`.
The chunk assemblers
`assemble_spectral_intensity_chunk` and
`assemble_spectral_intensity_bands_chunk` apply the same sampled-energy
Fermi occupation without materializing a `[K, B, E]` tensor.

```python
import jax
import jax.numpy as jnp

from diffpes.simul import evaluate_self_energy, spectral_intensity_eigen
from diffpes.types import make_self_energy_model

omega = jnp.linspace(-1.0, 1.0, 501)
sigma = evaluate_self_energy(
    omega,
    make_self_energy_model(gamma=0.08),
)
eigenvalues = jnp.array([-0.25, 0.30])
weights = jnp.array([0.8, 0.2])
intrinsic = jax.vmap(
    lambda energy, sigma_i: spectral_intensity_eigen(
        eigenvalues,
        weights,
        energy,
        sigma_i,
        1.0e-4,
    )
)(omega, sigma)
```

## Source Carriers and the Detector Boundary

`ArpesSpectrum` carries intensity, sampled energy, cumulative Cartesian path
length, every Cartesian path vector, and the registered sample-frame ID.
`ArpesCube` carries source-coordinate intensity on Cartesian $k_x$, $k_y$,
and energy axes. Neither carrier is a detector raster.

`simulate_arpes` and `simulate_arpes_cut` are the canonical typed drivers.
They build physical source carriers from mode-owned Hamiltonian or bulk-model
state, explicit matrix elements, and a causal self-energy. The shared detector
chain maps every source domain into explicit `DetectorCalibration` bins. It
mixes domains in detector space, applies transmission and native resolution,
and returns expected counts as a `DetectorRaster`.
Diffpes intentionally exposes no level-string or projection-probability
compatibility workflow.

Their keyword-only `kz_mode` extension selects exactly one of four routes.
These are the retained `native_direct` source, exact finite-energy
`bulk_direct`, wrapped finite-width `bulk_kz`, and a `coherent_slab`
depth-amplitude sum. The two escape-depth models are mutually exclusive.
`simulate_hv_scan` exposes the corresponding single-domain, pre-detector hν
stack; `hv_map_at_energy` makes a path-by-hν slice. See [kz Broadening and
Photon-Energy Scans](kz-broadening-and-photon-energy-scans.md) for carrier
rules, the finite-$\omega$ center, and the registered node budget.

Domain angles use the active right-handed z-y-z convention. The complete
sample-to-laboratory rotation applies sample azimuth after the domain
rotation. Cube targets that cross source-support faces are accepted only when
the projected map is signed diagonal or antidiagonal. A general projected
rotation must keep the complete inverse target strictly inside every source
exterior face. Otherwise, eager and compiled calls reject it. Gradients for a
general rotation cover only that fixed, smoothly enclosed chart.

An `ArpesSpectrum` is a line density already integrated over one declared
transverse slit aperture. Its forward `u` coordinate must be strictly
monotone. Its path must stay inside the single `v` bin. The conservative map
includes `abs(ds/du)` divided by that aperture width. A general path is never
promoted to an unstated two-dimensional density.

The frozen RM-2 Chinook fixture is deliberately narrower than this production
surface. It authenticates one complete `241 x 601` single-kz cut and applies a
test-only adapter that matches Chinook's sampled Gaussian response to a shared
pre-resolution input. This is K-only response compatibility: the production
long-tail helper is diagnostic, and the comparison makes no source-assembly,
detector-ordering, conservation, or absolute-scale claim.

## Choosing an Interface

- Use the resolvent path at degeneracies or whenever full source vectors must
  remain explicit.
- Use the eigen path only with gauge-invariant band weights and a justified
  isolated-band/group regime.
- Use `evaluate_self_energy` for every causal linewidth model.
- Use `load_vasp_context` and `prepare_projection` for input-boundary work.
- Use `run_vasp_workflow` only with an explicit phase-complete Hamiltonian and
  the complete coherent driver carriers. Parsed PROCAR weights remain
  phase-dead metadata.

See [Matrix Elements and Polarization](matrix-elements-and-polarization.md)
for the amplitude convention. See [Spectral Broadening and
Self-Energy](spectral-broadening-and-self-energy.md) for the causal contract.
See [kz Broadening and Photon-Energy
Scans](kz-broadening-and-photon-energy-scans.md) for bulk averaging and hν
outputs. See [Matrix-element
sensitivity](../tutorials/matrix-element-sensitivity.md) for complete-group
derivatives.
