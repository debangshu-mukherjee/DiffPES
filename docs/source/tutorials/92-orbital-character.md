# Orbital Character Maps

Orbital-resolved band weights from the PdTe2 four-layer PROCAR. The
panels color the band structure by orbital family, atomic species, and
layer. The final images fold the weights into ARPES-style fat-band maps.
The notebook reads the local `data/DFT` tree.

## Load the Public API

The structure and eigenvalue readers supply the geometry, the path, and
the Fermi reference. The projection reader returns the spin-orbit
carrier with orbital weights and spin channels.


```python
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from diffpes.inout import read_eigenval, read_poscar, read_procar
```

    WARNING:2026-08-13 18:43:55,006:jax._src.xla_bridge:876: An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.


## Load the Path and the Geometry

The species list orders the projection ions. The fractional atom heights
separate the top and bottom halves of the slab.


```python
DATA_ROOT = Path("..") / "data" / "DFT"
PDTE2_MGM_DIR = DATA_ROOT / "PdTe2" / "4ML" / "Output" / "MGM"
pdte2_fermi_ev = float(
    next(
        line
        for line in open(
            PDTE2_MGM_DIR / "OAM DATA" / "OUTCAR", encoding="utf-8"
        )
        if "E-fermi" in line
    ).split()[2]
)
pdte2_geo = read_poscar(
    str(DATA_ROOT / "PdTe2" / "4ML" / "PdTe2_4ML_0x_0y_0z.vasp")
)
pdte2_bands = read_eigenval(
    str(PDTE2_MGM_DIR / "EIGENVAL"), fermi_energy=pdte2_fermi_ev
)
band_shift = np.asarray(pdte2_bands.eigenvalues) - pdte2_fermi_ev
kcart = np.asarray(pdte2_bands.kpoints) @ np.asarray(pdte2_geo.reciprocal)
path_step = np.linalg.norm(np.diff(kcart, axis=0), axis=1)
path_dist = np.concatenate(([0.0], np.cumsum(path_step)))
gamma_index = int(
    np.argmin(np.linalg.norm(np.asarray(pdte2_bands.kpoints), axis=1))
)
path_axis = path_dist - path_dist[gamma_index]
species = np.asarray(pdte2_geo.species)
heights = np.asarray(pdte2_geo.positions)[:, 2]
print("species:", list(species))
print("fractional heights:", np.round(heights, 3))
```

    species: [np.str_('Pd'), np.str_('Pd'), np.str_('Pd'), np.str_('Pd'), np.str_('Te'), np.str_('Te'), np.str_('Te'), np.str_('Te'), np.str_('Te'), np.str_('Te'), np.str_('Te'), np.str_('Te')]
    fractional heights: [0.313 0.438 0.563 0.688 0.281 0.406 0.531 0.656 0.344 0.469 0.594 0.719]


## Read the Projection Tables

The reader returns the spin-orbit projection carrier: one charge table
and six nonnegative spin channels. The weight array covers every
k-point, band, ion, and orbital on the path.


```python
projection = read_procar(
    str(PDTE2_MGM_DIR / "PROCAR"), return_mode="full"
)
weights = np.asarray(projection.projections)
spin_channels = np.asarray(projection.spin)
print("weight block:", weights.shape)
print("spin block:", spin_channels.shape)
print(
    "mean state weight:",
    round(float(weights.sum(axis=(2, 3)).mean()), 3),
)
```

    weight block: (400, 144, 12, 9)
    spin block: (400, 144, 12, 6)
    mean state weight: 0.611


## Group the Weights

The orbital columns follow the VASP order: s, then three p, then five d.
The species masks split palladium from tellurium. The height masks split
the top half of the slab from the bottom half.


```python
total_weight = weights.sum(axis=(2, 3))
safe_total = np.where(total_weight > 1.0e-12, total_weight, 1.0)
pd_mask = species == "Pd"
te_mask = species == "Te"
top_mask = heights > np.median(heights)
pd_d_fraction = weights[:, :, pd_mask, 4:9].sum(axis=(2, 3)) / safe_total
te_p_fraction = weights[:, :, te_mask, 1:4].sum(axis=(2, 3)) / safe_total
p_total = weights[:, :, :, 1:4].sum(axis=(2, 3))
pz_fraction = weights[:, :, :, 2].sum(axis=2) / np.where(
    p_total > 1.0e-12, p_total, 1.0
)
surface_fraction = weights[:, :, top_mask, :].sum(axis=(2, 3)) / safe_total
fig, ax = plt.subplots(figsize=(6.2, 4.0))
ax.hist(total_weight.ravel(), bins=60, color="tab:gray")
ax.set_xlabel("summed projection weight per state")
ax.set_ylabel("state count")
ax.set_title("projection completeness across the path")
plt.show()
```



![png](92-orbital-character_files/92-orbital-character_8_0.png)



## Color the Bands by Character

Each scatter colors the same path by one fraction. The palladium d
weight dominates the deeper valence manifold. The tellurium p weight
carries the states near the Fermi level.


```python
path_mesh = np.repeat(path_axis[:, None], band_shift.shape[1], axis=1)
window = (band_shift > -3.0) & (band_shift < 1.0)
fig, ax = plt.subplots(figsize=(6.8, 4.8))
points = ax.scatter(
    path_mesh[window],
    band_shift[window],
    c=pd_d_fraction[window],
    s=2.0,
    cmap="viridis",
    vmin=0.0,
    vmax=1.0,
)
ax.axhline(0.0, color="0.4", linewidth=0.8)
ax.set_xlabel(r"$k - k_\Gamma$ ($\AA^{-1}$)")
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_title("palladium d fraction along M--Gamma--M")
fig.colorbar(points, ax=ax, label="Pd d fraction")
plt.show()
```



![png](92-orbital-character_files/92-orbital-character_10_0.png)




```python
fig, ax = plt.subplots(figsize=(6.8, 4.8))
points = ax.scatter(
    path_mesh[window],
    band_shift[window],
    c=te_p_fraction[window],
    s=2.0,
    cmap="plasma",
    vmin=0.0,
    vmax=1.0,
)
ax.axhline(0.0, color="0.4", linewidth=0.8)
ax.set_xlabel(r"$k - k_\Gamma$ ($\AA^{-1}$)")
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_title("tellurium p fraction along M--Gamma--M")
fig.colorbar(points, ax=ax, label="Te p fraction")
plt.show()
```



![png](92-orbital-character_files/92-orbital-character_11_0.png)




```python
fig, ax = plt.subplots(figsize=(6.8, 4.8))
points = ax.scatter(
    path_mesh[window],
    band_shift[window],
    c=pz_fraction[window],
    s=2.0,
    cmap="coolwarm",
    vmin=0.0,
    vmax=1.0,
)
ax.axhline(0.0, color="0.4", linewidth=0.8)
ax.set_xlabel(r"$k - k_\Gamma$ ($\AA^{-1}$)")
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_title("out-of-plane share of the p weight")
fig.colorbar(points, ax=ax, label=r"$p_z$ / $p$ fraction")
plt.show()
```



![png](92-orbital-character_files/92-orbital-character_12_0.png)




```python
fig, ax = plt.subplots(figsize=(6.8, 4.8))
points = ax.scatter(
    path_mesh[window],
    band_shift[window],
    c=surface_fraction[window],
    s=2.0,
    cmap="magma",
    vmin=0.0,
    vmax=1.0,
)
ax.axhline(0.0, color="0.4", linewidth=0.8)
ax.set_xlabel(r"$k - k_\Gamma$ ($\AA^{-1}$)")
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_title("top-half weight along the path")
fig.colorbar(points, ax=ax, label="top-half fraction")
plt.show()
```



![png](92-orbital-character_files/92-orbital-character_13_0.png)



## Decompose the Gamma-Point States

The stacked bars split every Gamma state in the window into s, p, and d
weight. The valence top mixes tellurium p with palladium d.


```python
gamma_window = (band_shift[gamma_index] > -2.0) & (
    band_shift[gamma_index] < 0.5
)
gamma_band_indices = np.nonzero(gamma_window)[0]
gamma_energies = band_shift[gamma_index, gamma_band_indices]
gamma_s = weights[gamma_index, gamma_band_indices, :, 0].sum(axis=1)
gamma_p = weights[gamma_index, gamma_band_indices, :, 1:4].sum(axis=(1, 2))
gamma_d = weights[gamma_index, gamma_band_indices, :, 4:9].sum(axis=(1, 2))
gamma_norm = np.where(
    gamma_s + gamma_p + gamma_d > 1.0e-12,
    gamma_s + gamma_p + gamma_d,
    1.0,
)
bar_positions = np.arange(gamma_band_indices.shape[0], dtype=np.float64)
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.bar(bar_positions, gamma_s / gamma_norm, label="s")
ax.bar(
    bar_positions,
    gamma_p / gamma_norm,
    bottom=gamma_s / gamma_norm,
    label="p",
)
ax.bar(
    bar_positions,
    gamma_d / gamma_norm,
    bottom=(gamma_s + gamma_p) / gamma_norm,
    label="d",
)
ax.set_xticks(bar_positions[:: max(1, bar_positions.shape[0] // 12)])
ax.set_xticklabels(
    [
        f"{energy:.2f}"
        for energy in gamma_energies[
            :: max(1, bar_positions.shape[0] // 12)
        ]
    ],
    rotation=45,
)
ax.set_xlabel(r"Gamma-state energy $E - E_F$ (eV)")
ax.set_ylabel("orbital share")
ax.set_title("orbital composition of the Gamma states")
ax.legend()
plt.show()
```



![png](92-orbital-character_files/92-orbital-character_15_0.png)



## Fold the Weights into Fat-Band Maps

Each state contributes one Lorentzian line scaled by its weight. A Fermi
factor at 100 K removes the unoccupied side. The three maps show the
palladium d channel, the tellurium p channel, and their difference.


```python
omega_ev = jnp.linspace(-3.0, 0.5, 241)
gamma_width_ev = 0.035
occupation = 1.0 / (1.0 + jnp.exp(omega_ev / 0.0086))
levels = jnp.asarray(band_shift)
lorentzians = (gamma_width_ev / jnp.pi) / (
    (omega_ev[None, None, :] - levels[:, :, None]) ** 2
    + gamma_width_ev**2
)
pd_d_map = (
    (lorentzians * jnp.asarray(pd_d_fraction)[:, :, None]).sum(axis=1)
    * occupation[None, :]
).T
fig, ax = plt.subplots(figsize=(6.6, 4.8))
image = ax.imshow(
    np.asarray(pd_d_map),
    origin="lower",
    aspect="auto",
    extent=(
        float(path_axis[0]),
        float(path_axis[-1]),
        float(omega_ev[0]),
        float(omega_ev[-1]),
    ),
    cmap="viridis",
)
ax.set_xlabel(r"$k - k_\Gamma$ ($\AA^{-1}$)")
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_title("palladium d fat-band map")
fig.colorbar(image, ax=ax, label="weighted intensity")
plt.show()
```



![png](92-orbital-character_files/92-orbital-character_17_0.png)




```python
te_p_map = (
    (lorentzians * jnp.asarray(te_p_fraction)[:, :, None]).sum(axis=1)
    * occupation[None, :]
).T
fig, ax = plt.subplots(figsize=(6.6, 4.8))
image = ax.imshow(
    np.asarray(te_p_map),
    origin="lower",
    aspect="auto",
    extent=(
        float(path_axis[0]),
        float(path_axis[-1]),
        float(omega_ev[0]),
        float(omega_ev[-1]),
    ),
    cmap="plasma",
)
ax.set_xlabel(r"$k - k_\Gamma$ ($\AA^{-1}$)")
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_title("tellurium p fat-band map")
fig.colorbar(image, ax=ax, label="weighted intensity")
plt.show()
```



![png](92-orbital-character_files/92-orbital-character_18_0.png)




```python
difference_map = np.asarray(pd_d_map) - np.asarray(te_p_map)
difference_scale = float(np.abs(difference_map).max())
fig, ax = plt.subplots(figsize=(6.6, 4.8))
image = ax.imshow(
    difference_map,
    origin="lower",
    aspect="auto",
    extent=(
        float(path_axis[0]),
        float(path_axis[-1]),
        float(omega_ev[0]),
        float(omega_ev[-1]),
    ),
    cmap="RdBu_r",
    vmin=-difference_scale,
    vmax=difference_scale,
)
ax.set_xlabel(r"$k - k_\Gamma$ ($\AA^{-1}$)")
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_title("d minus p channel contrast")
fig.colorbar(image, ax=ax, label="intensity difference")
plt.show()
```



![png](92-orbital-character_files/92-orbital-character_19_0.png)




```python
surface_map = (
    (lorentzians * jnp.asarray(surface_fraction)[:, :, None]).sum(axis=1)
    * occupation[None, :]
).T
fig, ax = plt.subplots(figsize=(6.6, 4.8))
image = ax.imshow(
    np.asarray(surface_map),
    origin="lower",
    aspect="auto",
    extent=(
        float(path_axis[0]),
        float(path_axis[-1]),
        float(omega_ev[0]),
        float(omega_ev[-1]),
    ),
    cmap="magma",
)
ax.set_xlabel(r"$k - k_\Gamma$ ($\AA^{-1}$)")
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_title("top-half weighted fat-band map")
fig.colorbar(image, ax=ax, label="weighted intensity")
plt.show()
```



![png](92-orbital-character_files/92-orbital-character_20_0.png)



## Map the Spin Texture

The signed z spin is the fifth channel minus the sixth channel, summed
over ions. Red and blue mark opposite spin signs. Opposite momenta carry
opposite signs across Gamma.


```python
sz_signed = (
    spin_channels[:, :, :, 4] - spin_channels[:, :, :, 5]
).sum(axis=2)
sz_scale = float(np.abs(sz_signed).max())
fig, ax = plt.subplots(figsize=(6.8, 4.8))
points = ax.scatter(
    path_mesh[window],
    band_shift[window],
    c=sz_signed[window],
    s=2.0,
    cmap="RdBu_r",
    vmin=-sz_scale,
    vmax=sz_scale,
)
ax.axhline(0.0, color="0.4", linewidth=0.8)
ax.set_xlabel(r"$k - k_\Gamma$ ($\AA^{-1}$)")
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_title("signed z spin along M--Gamma--M")
fig.colorbar(points, ax=ax, label="signed z spin weight")
plt.show()
```



![png](92-orbital-character_files/92-orbital-character_22_0.png)
