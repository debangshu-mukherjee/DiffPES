---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Quickstart: Two Incoherent Spectrum Tiers

Build a synthetic band map, compare the two retained projection tiers, and
differentiate a spectral observable. These tiers consume orbital
probabilities. The separate coherent pipeline is introduced in
[Matrix Elements and Polarization](../guides/matrix-elements-and-polarization.md).

```{code-cell} ipython3
import diffpes
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

print(f"diffpes {diffpes.__version__}")
print(f"x64 enabled: {jax.config.jax_enable_x64}")
```

## Synthetic Bands and Projections

Use two cosine bands on a one-dimensional path. `eigenbands` has shape
`[K, B]`. `surface_orb` has the VASP projection shape `[K, B, A, 9]`, with
the final axis ordered as
`s, py, pz, px, dxy, dyz, dz2, dxz, dx2-y2`.

```{code-cell} ipython3
nkpt = 120
kpath = jnp.linspace(-jnp.pi, jnp.pi, nkpt)
eigenbands = jnp.stack(
    [
        -1.2 - 0.8 * jnp.cos(kpath),
        -0.4 - 0.9 * jnp.cos(kpath + 0.3),
    ],
    axis=1,
)

surface_orb = jnp.zeros((nkpt, 2, 1, 9))
surface_orb = surface_orb.at[:, 0, 0, 2].set(
    jnp.cos(kpath / 2.0) ** 2
)
surface_orb = surface_orb.at[:, 0, 0, 3].set(
    jnp.sin(kpath / 2.0) ** 2
)
surface_orb = surface_orb.at[:, 1, 0, 4].set(0.7)
surface_orb = surface_orb.at[:, 1, 0, 8].set(0.3)

print(eigenbands.shape, surface_orb.shape)
```

## Uniform Projection Spectrum

The `novice` tier sums non-s projection probabilities, applies
Fermi--Dirac occupation, and broadens each band with a Voigt profile.

```{code-cell} ipython3
spectrum_novice = diffpes.simul.simulate_expanded(
    level="novice",
    eigenbands=eigenbands,
    surface_orb=surface_orb,
    ef=0.0,
    sigma=0.06,
    gamma=0.08,
    fidelity=1200,
    temperature=30.0,
)
print(spectrum_novice.intensity.shape)
```

## Yeh--Lindau Projection Spectrum

The `basic` tier uses element- and subshell-resolved Yeh--Lindau cross
sections. It requires one atom-major basis row for every projection channel.
This synthetic example labels the atom as scandium. Its nine channels use the
$3s$, $3p$, and $3d$ subshells, which have table support at 80 eV.

```{code-cell} ipython3
basis = diffpes.types.make_orbital_basis(
    atom_indices=(0,) * 9,
    n=(3,) * 9,
    l=(0, 1, 1, 1, 2, 2, 2, 2, 2),
    m=(0, -1, 0, 1, -2, -1, 0, 1, 2),
)
spectrum_basic = diffpes.simul.simulate_expanded(
    level="basic",
    eigenbands=eigenbands,
    surface_orb=surface_orb,
    ef=0.0,
    sigma=0.06,
    fidelity=1200,
    temperature=30.0,
    photon_energy=80.0,
    basis=basis,
    atomic_numbers=(21,),
)
print(spectrum_basic.intensity.shape)
```

The two spectra have identical dispersions but different relative
brightness. The basic tier changes p- and d-derived intensities through
isolated-atom cross sections. Neither tier has the complex coefficient phase
needed for polarization selection rules or atomic-centre interference.

```{code-cell} ipython3
extent = [
    float(kpath[0]),
    float(kpath[-1]),
    float(spectrum_novice.energy_axis[0]),
    float(spectrum_novice.energy_axis[-1]),
]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
for ax, spectrum, title in zip(
    axes,
    (spectrum_novice, spectrum_basic),
    ("novice: uniform", "basic: Yeh--Lindau"),
):
    ax.imshow(
        spectrum.intensity.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="inferno",
    )
    ax.set_title(title)
    ax.set_xlabel("path coordinate")
axes[0].set_ylabel(r"$E-E_F$ (eV)")
plt.show()
```

## Differentiate a Spectral Observable

Continuous array and scalar inputs remain differentiable. Here the observable
is the total novice intensity in a window below the Fermi level.

```{code-cell} ipython3
def window_intensity(sigma):
    spectrum = diffpes.simul.simulate_expanded(
        level="novice",
        eigenbands=eigenbands,
        surface_orb=surface_orb,
        sigma=sigma,
        gamma=0.08,
        fidelity=1200,
        temperature=30.0,
    )
    selected = (
        (spectrum.energy_axis > -0.4)
        & (spectrum.energy_axis < 0.0)
    )
    return jnp.sum(spectrum.intensity * selected)


gradient = jax.grad(window_intensity)(0.06)
print(f"d window intensity / d sigma = {float(gradient):.3f}")
```

## Next Steps

- Read [Simulation Tiers and the Coherent Pipeline](../guides/simulation-levels.md).
- Work through the executable
  [matrix-element sensitivity tutorial](matrix-element-sensitivity.md).
- Use the [VASP ingestion guide](../guides/vasp-data-ingestion.md) with real
  electronic-structure output.
