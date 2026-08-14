# Quickstart: Coherent Spectral Assembly

Build an intrinsic ARPES spectrum from gauge-invariant matrix-element
weights, a causal self-energy, and sampled-energy Fermi occupation. The
degeneracy-safe resolvent path is introduced alongside the faster eigen path.

The final section maps this intrinsic observable into explicit native detector
bins and applies the same effects chain used by the canonical coherent driver.

```python
import diffpes
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

print(f"diffpes {diffpes.__version__}")
print(f"x64 enabled: {jax.config.jax_enable_x64}")
```

## Synthetic Band and Matrix-Element Data

Use two dispersing eigenvalues and positive gauge-invariant band weights on a
Cartesian sample-frame path. In a full calculation, the weights come from
coherent matrix-element channels after polarization contraction and band
projection.

```python
nkpt = 120
k_cart_x = jnp.linspace(-1.0, 1.0, nkpt)
kpoints_cart_inv_ang = jnp.stack(
    (k_cart_x, jnp.zeros_like(k_cart_x), jnp.zeros_like(k_cart_x)),
    axis=1,
)
k_axis = k_cart_x - k_cart_x[0]
eigenvalues = jnp.stack(
    (
        -0.65 - 0.45 * jnp.cos(jnp.pi * k_cart_x),
        0.15 + 0.35 * jnp.cos(jnp.pi * k_cart_x + 0.2),
    ),
    axis=1,
)
band_weights = jnp.stack(
    (
        0.2 + 0.8 * jnp.cos(0.5 * jnp.pi * k_cart_x) ** 2,
        0.3 + 0.5 * jnp.sin(0.5 * jnp.pi * k_cart_x) ** 2,
    ),
    axis=1,
)

print(eigenvalues.shape, band_weights.shape)
```

## Assemble the Occupied Intrinsic Spectrum

Evaluate a retarded constant-linewidth self-energy, sample the eigen spectral
primitive, and apply Fermi occupation on the same energy nodes.

```python
omega = jnp.linspace(-1.5, 0.8, 500)
self_energy_model = diffpes.types.make_self_energy_model(gamma=0.08)
sigma_omega = diffpes.simul.evaluate_self_energy(omega, self_energy_model)

def one_k_spectrum(eigenvalues_k, weights_k):
    return jax.vmap(
        lambda energy, sigma: diffpes.simul.spectral_intensity_eigen(
            eigenvalues_k,
            weights_k,
            energy,
            sigma,
            1.0e-4,
        )
    )(omega, sigma_omega)

intrinsic = jax.vmap(one_k_spectrum)(eigenvalues, band_weights)
occupation = jax.vmap(
    lambda energy: diffpes.simul.fermi_dirac(energy, 0.0, 30.0)
)(omega)
spectrum = diffpes.types.make_arpes_spectrum(
    intensity=intrinsic * occupation[None, :],
    energy_axis=omega,
    k_axis=k_axis,
    kpoints_cart_inv_ang=kpoints_cart_inv_ang,
)
print(spectrum.intensity.shape)
```

The full Cartesian vectors remain attached to the spectrum. Equal cumulative
path lengths therefore cannot conflate paths in different directions.

```python
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.imshow(
    spectrum.intensity.T,
    origin="lower",
    aspect="auto",
    extent=(
        float(k_cart_x[0]),
        float(k_cart_x[-1]),
        float(omega[0]),
        float(omega[-1]),
    ),
    cmap="inferno",
)
ax.set_xlabel(r"$k_x$ ($\AA^{-1}$)")
ax.set_ylabel(r"$E-E_F$ (eV)")
ax.set_title("occupied intrinsic spectral intensity")
plt.show()
```

## Use the Degeneracy-Safe Resolvent

At exact or near degeneracy, keep the Hamiltonian and full transition source
instead of differentiating eigenvectors.

```python
hamiltonian = jnp.array(
    [[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]],
    dtype=jnp.complex128,
)
transition_sources = jnp.array([[1.0 + 0.0j, 0.5 + 0.2j]])
resolvent_value = diffpes.simul.spectral_intensity_resolvent(
    hamiltonian,
    transition_sources,
    jnp.asarray(0.0),
    jnp.asarray(-0.08j),
    1.0e-4,
)
print(float(resolvent_value))
```

## Map into Native Detector Counts

Attach explicit experiment, calibration, and nuisance state. The public
effects chain maps the self-describing source path into native angular and
recorded-energy bins before applying transmission, resolution, background,
sensitivity, exposure, and bin-volume conversion.

```python
experiment = diffpes.types.make_experiment_geometry(
    photon_energy_ev=50.0,
    polarization=jnp.array([1.0 + 0.0j, 0.0j, 0.0j]),
    work_function_ev=4.0,
    temperature_k=30.0,
    slit="H",
)
calibration = diffpes.types.make_detector_calibration(
    u_bin_edges=jnp.linspace(-0.24, 0.24, 49),
    v_bin_edges=jnp.array([-0.04, 0.04]),
    energy_bin_edges_ev=jnp.linspace(-1.35, 0.65, 81),
    psf_fwhm_u=0.012,
    psf_fwhm_v=0.010,
    psf_fwhm_energy_ev=0.040,
    transmission_reference_domain_ev=jnp.array([44.0, 47.0]),
)
effects = diffpes.types.make_detector_effects(
    domain_logits=jnp.array([0.0]),
    domain_euler_angles_rad=jnp.zeros((1, 3)),
    transmission_raw_slopes=jnp.array([-0.4, 0.2]),
    background_coefficients=jnp.array([-8.0]),
    sensitivity_coefficients=jnp.array([]),
    exposure=100.0,
    background_mode="flat",
    sensitivity_mode="constant",
    domain_frame_ids=("org.diffpes.frame.sample_cartesian",),
)
detector = diffpes.simul.apply_detector_effects(
    (spectrum,), experiment, calibration, effects
)
print(detector.expected_counts.shape, detector.channel_labels)
```

The result stays on native detector coordinates. It is not relabeled as a
Cartesian momentum raster.

```python
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.imshow(
    detector.expected_counts[0, :, 0, :].T,
    origin="lower",
    aspect="auto",
    extent=(
        float(detector.detector_u_axis[0]),
        float(detector.detector_u_axis[-1]),
        float(detector.energy_axis[0]),
        float(detector.energy_axis[-1]),
    ),
    cmap="magma",
)
ax.set_xlabel(r"native detector $u$ (rad)")
ax.set_ylabel(r"recorded $E-E_F$ (eV)")
ax.set_title("expected detector counts")
plt.show()
```

## Differentiate a Spectral Observable

The self-energy coordinates remain differentiable through the sampled
spectral assembly.

```python
def occupied_weight(gamma):
    model = diffpes.types.make_self_energy_model(gamma=gamma)
    sigma = diffpes.simul.evaluate_self_energy(omega, model)
    intensity = jax.vmap(
        lambda energy, sigma_i: diffpes.simul.spectral_intensity_eigen(
            eigenvalues[0],
            band_weights[0],
            energy,
            sigma_i,
            1.0e-4,
        )
    )(omega, sigma)
    selected = (omega > -0.7) & (omega < 0.0)
    return jnp.sum(intensity * occupation * selected)

gradient = jax.grad(occupied_weight)(0.08)
print(f"d occupied weight / d gamma = {float(gradient):.3f}")
```

## Next Steps

- Read [Simulating ARPES Spectra](../guides/simulating-arpes-spectra.md).
- Work through the executable
  [matrix-element sensitivity tutorial](matrix-element-sensitivity.md).
- Use the [VASP ingestion guide](../guides/vasp-data-ingestion.md) to prepare
  real electronic-structure inputs.
