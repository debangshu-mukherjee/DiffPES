# Anchor Material Band Structures

Band structures from the internal DFT calculations of Bi2Se3, PdTe2, and
the bismuth telluride family. The final panels turn the same eigenvalues
into ARPES-style intensity maps through the spectral assembly. The
notebook reads the local `data/DFT` tree.

## Load the Public API

The readers return validated carriers for eigenvalues, Fermi levels,
lattices, and k-point paths. The k-space calls build momentum axes. The
spectral call folds eigenvalues into occupied intensity maps.


```python
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from diffpes.inout import (
    dedupe_band_path,
    plot_arpes_with_kpath,
    read_doscar,
    read_eigenval,
    read_kpoints,
    read_outcar,
    read_poscar,
)
from diffpes.simul import assemble_spectral_intensity_bands_chunk
from diffpes.tightb import kpath_arc_length, kpoints_frac_to_cart
from diffpes.types import (
    make_arpes_spectrum,
    make_kpath,
    make_self_energy_model,
)
```

## Locate the Local Data Tree

Every path is relative to the tutorials directory. The data tree sits one
level up, under `data/DFT`.


```python
DATA_ROOT = Path("..") / "data" / "DFT"
BI2SE3_DIR = DATA_ROOT / "Bi2Se3" / "6QL" / "Output few bands"
PDTE2_DIR = DATA_ROOT / "PdTe2" / "4ML" / "Output"
BIXTEY_DIR = DATA_ROOT / "BixTey"
print("data tree present:", DATA_ROOT.is_dir())
```

    data tree present: True


## Read the Fermi Levels

`read_outcar` returns the Fermi energy and electron count of each
calculation. The Bi2Se3 value comes from the self-consistent OUTCAR. The
PdTe2 value comes from the retained band-run OUTCAR.


```python
bi2se3_summary = read_outcar(str(BI2SE3_DIR / "OUTCAR_SCF"))
pdte2_summary = read_outcar(
    str(PDTE2_DIR / "MGM" / "OAM DATA" / "OUTCAR")
)
bi2se3_fermi_ev = float(bi2se3_summary.fermi_energy)
pdte2_fermi_ev = float(pdte2_summary.fermi_energy)
print("Bi2Se3 Fermi (eV):", bi2se3_fermi_ev)
print("Bi2Se3 NELECT:", float(bi2se3_summary.nelect))
print("PdTe2 Fermi (eV):", pdte2_fermi_ev)
print("PdTe2 NELECT:", float(pdte2_summary.nelect))
```

    Bi2Se3 Fermi (eV): 0.0178
    Bi2Se3 NELECT: 168.0
    PdTe2 Fermi (eV): 2.1733
    PdTe2 NELECT: 88.0


## Load the Anchor Band Paths

Each path is a line-mode calculation. The eigenvalue reader returns
absolute eigenvalues. The KPOINTS reader keeps the symmetry labels for
the intensity maps. Every plot subtracts the Fermi energy.


```python
bi2se3_geo = read_poscar(str(BI2SE3_DIR / "POSCAR"))
pdte2_geo = read_poscar(
    str(DATA_ROOT / "PdTe2" / "4ML" / "PdTe2_4ML_0x_0y_0z.vasp")
)
bi2se3_bands = read_eigenval(
    str(BI2SE3_DIR / "MGM" / "EIGENVAL"), fermi_energy=bi2se3_fermi_ev
)
pdte2_mgm = read_eigenval(
    str(PDTE2_DIR / "MGM" / "EIGENVAL"), fermi_energy=pdte2_fermi_ev
)
pdte2_kgk = read_eigenval(
    str(PDTE2_DIR / "KGK" / "EIGENVAL"), fermi_energy=pdte2_fermi_ev
)
bi2se3_kinfo = read_kpoints(str(BI2SE3_DIR / "MGM" / "KPOINTS"))
pdte2_mgm_kinfo = read_kpoints(str(PDTE2_DIR / "MGM" / "KPOINTS"))
print("Bi2Se3 eigenvalue block:", bi2se3_bands.eigenvalues.shape)
print("PdTe2 MGM eigenvalue block:", pdte2_mgm.eigenvalues.shape)
print("PdTe2 KGK eigenvalue block:", pdte2_kgk.eigenvalues.shape)
print("Bi2Se3 path labels:", bi2se3_kinfo.labels)
```

    Bi2Se3 eigenvalue block: (400, 176)
    PdTe2 MGM eigenvalue block: (400, 144)
    PdTe2 KGK eigenvalue block: (400, 144)
    Bi2Se3 path labels: ('M', 'G', 'M')


## Convert the Paths to Inverse Angstrom

`make_kpath` wraps each fractional path. `kpath_arc_length` returns the
cumulative Cartesian distance through the reciprocal lattice. The signed
path coordinate is zero at the Gamma point.


```python
bi2se3_dist = kpath_arc_length(
    make_kpath(bi2se3_bands.kpoints), bi2se3_geo
)
bi2se3_gamma = int(
    np.argmin(np.linalg.norm(np.asarray(bi2se3_bands.kpoints), axis=1))
)
bi2se3_axis = np.asarray(bi2se3_dist - bi2se3_dist[bi2se3_gamma])
pdte2_mgm_dist = kpath_arc_length(
    make_kpath(pdte2_mgm.kpoints), pdte2_geo
)
pdte2_mgm_gamma = int(
    np.argmin(np.linalg.norm(np.asarray(pdte2_mgm.kpoints), axis=1))
)
pdte2_mgm_axis = np.asarray(
    pdte2_mgm_dist - pdte2_mgm_dist[pdte2_mgm_gamma]
)
pdte2_kgk_dist = kpath_arc_length(
    make_kpath(pdte2_kgk.kpoints), pdte2_geo
)
pdte2_kgk_gamma = int(
    np.argmin(np.linalg.norm(np.asarray(pdte2_kgk.kpoints), axis=1))
)
pdte2_kgk_axis = np.asarray(
    pdte2_kgk_dist - pdte2_kgk_dist[pdte2_kgk_gamma]
)
print("Bi2Se3 path half width (1/Ang):", float(bi2se3_axis.max()))
print("PdTe2 MGM path half width (1/Ang):", float(pdte2_mgm_axis.max()))
```

    Bi2Se3 path half width (1/Ang): 0.8654930508183045
    PdTe2 MGM path half width (1/Ang): 0.9005955408277536


## Plot the Bi2Se3 Slab Bands

The full window shows the slab valence manifold. The zoom shows the
topological surface state. The Dirac crossing sits close to the Fermi
level.


```python
bi2se3_shift = np.asarray(bi2se3_bands.eigenvalues) - bi2se3_fermi_ev
fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.plot(bi2se3_axis, bi2se3_shift, color="tab:blue", linewidth=0.4)
ax.axhline(0.0, color="0.4", linewidth=0.8)
ax.set_xlabel(r"$k - k_\Gamma$ along M--$\Gamma$--M ($\AA^{-1}$)")
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_ylim(-6.0, 2.0)
ax.set_title("Bi2Se3 six-layer slab bands")
plt.show()
```



![png](91-anchor-band-structures_files/91-anchor-band-structures_12_0.png)




```python
fig, ax = plt.subplots(figsize=(6.0, 4.6))
ax.plot(bi2se3_axis, bi2se3_shift, color="tab:blue", linewidth=0.7)
ax.axhline(0.0, color="0.4", linewidth=0.8)
ax.set_xlim(-0.35, 0.35)
ax.set_ylim(-1.2, 0.7)
ax.set_xlabel(r"$k - k_\Gamma$ ($\AA^{-1}$)")
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_title("Bi2Se3 surface-state window")
plt.show()
```



![png](91-anchor-band-structures_files/91-anchor-band-structures_13_0.png)



## Plot the PdTe2 Slab Bands

Dense slab subbands fill the four-layer window. The zoom panels cover the
states near Gamma on both retained paths.


```python
pdte2_mgm_shift = np.asarray(pdte2_mgm.eigenvalues) - pdte2_fermi_ev
fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.plot(pdte2_mgm_axis, pdte2_mgm_shift, color="tab:red", linewidth=0.4)
ax.axhline(0.0, color="0.4", linewidth=0.8)
ax.set_xlabel(r"$k - k_\Gamma$ along M--$\Gamma$--M ($\AA^{-1}$)")
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_ylim(-6.0, 2.0)
ax.set_title("PdTe2 four-layer slab bands")
plt.show()
```



![png](91-anchor-band-structures_files/91-anchor-band-structures_15_0.png)




```python
fig, ax = plt.subplots(figsize=(6.0, 4.6))
ax.plot(pdte2_mgm_axis, pdte2_mgm_shift, color="tab:red", linewidth=0.7)
ax.axhline(0.0, color="0.4", linewidth=0.8)
ax.set_xlim(-0.45, 0.45)
ax.set_ylim(-1.6, 0.7)
ax.set_xlabel(r"$k - k_\Gamma$ ($\AA^{-1}$)")
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_title("PdTe2 near-Gamma window")
plt.show()
```



![png](91-anchor-band-structures_files/91-anchor-band-structures_16_0.png)




```python
pdte2_kgk_shift = np.asarray(pdte2_kgk.eigenvalues) - pdte2_fermi_ev
fig, ax = plt.subplots(figsize=(6.0, 4.6))
ax.plot(pdte2_kgk_axis, pdte2_kgk_shift, color="tab:red", linewidth=0.5)
ax.axhline(0.0, color="0.4", linewidth=0.8)
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(-2.0, 0.7)
ax.set_xlabel(r"$k - k_\Gamma$ along K--$\Gamma$--K ($\AA^{-1}$)")
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_title("PdTe2 K--Gamma--K window")
plt.show()
```



![png](91-anchor-band-structures_files/91-anchor-band-structures_17_0.png)



## Survey the Bismuth Telluride Family

Supplementary calculations cover BiTe, Bi2Te3, and Bi4Te3. Each panel
uses the momentum axis from its own structure file. Each Fermi value
comes from the matching band-run OUTCAR.


```python
bite_dir = BIXTEY_DIR / "BiTe" / "Bulk" / "Output"
bite_fermi_ev = float(
    read_outcar(str(bite_dir / "MGM" / "OUTCAR_BAND")).fermi_energy
)
bite_geo = read_poscar(str(bite_dir / "POSCAR"))
bite_bands = read_eigenval(
    str(bite_dir / "MGM" / "EIGENVAL"), fermi_energy=bite_fermi_ev
)
bi2te3_dir = BIXTEY_DIR / "Bi2Te3" / "Bulk" / "Output"
bi2te3_fermi_ev = float(
    read_outcar(str(bi2te3_dir / "KGK" / "OUTCAR_BAND")).fermi_energy
)
bi2te3_geo = read_poscar(str(bi2te3_dir / "POSCAR"))
bi2te3_bands = read_eigenval(
    str(bi2te3_dir / "KGK" / "EIGENVAL"), fermi_energy=bi2te3_fermi_ev
)
bi4te3_dir = BIXTEY_DIR / "Bi4Te3" / "Bulk" / "Output"
bi4te3_fermi_ev = float(
    read_outcar(str(bi4te3_dir / "MGM" / "OUTCAR_BAND")).fermi_energy
)
bi4te3_geo = read_poscar(str(bi4te3_dir / "POSCAR"))
bi4te3_bands = read_eigenval(
    str(bi4te3_dir / "MGM" / "EIGENVAL"), fermi_energy=bi4te3_fermi_ev
)
bi4te3_slab_dir = BIXTEY_DIR / "Bi4Te3" / "5BL 5QL" / "Output"
bi4te3_slab_fermi_ev = float(
    read_outcar(
        str(bi4te3_slab_dir / "MGM" / "OUTCAR_BAND")
    ).fermi_energy
)
bi4te3_slab_geo = read_poscar(str(bi4te3_slab_dir / "POSCAR"))
bi4te3_slab_bands = read_eigenval(
    str(bi4te3_slab_dir / "MGM" / "EIGENVAL"),
    fermi_energy=bi4te3_slab_fermi_ev,
)
print("BiTe bands:", bite_bands.eigenvalues.shape)
print("Bi2Te3 bands:", bi2te3_bands.eigenvalues.shape)
print("Bi4Te3 bulk bands:", bi4te3_bands.eigenvalues.shape)
print("Bi4Te3 slab bands:", bi4te3_slab_bands.eigenvalues.shape)
```

    BiTe bands: (400, 140)
    Bi2Te3 bands: (400, 140)
    Bi4Te3 bulk bands: (400, 140)
    Bi4Te3 slab bands: (400, 200)



```python
bite_axis = np.asarray(
    kpath_arc_length(make_kpath(bite_bands.kpoints), bite_geo)
)
bite_shift = np.asarray(bite_bands.eigenvalues) - bite_fermi_ev
fig, ax = plt.subplots(figsize=(6.0, 4.4))
ax.plot(bite_axis, bite_shift, color="tab:purple", linewidth=0.4)
ax.axhline(0.0, color="0.4", linewidth=0.8)
ax.set_ylim(-4.0, 2.0)
ax.set_xlabel(r"M--$\Gamma$--M path distance ($\AA^{-1}$)")
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_title("BiTe bulk bands")
plt.show()
```



![png](91-anchor-band-structures_files/91-anchor-band-structures_20_0.png)




```python
bi2te3_axis = np.asarray(
    kpath_arc_length(make_kpath(bi2te3_bands.kpoints), bi2te3_geo)
)
bi2te3_shift = np.asarray(bi2te3_bands.eigenvalues) - bi2te3_fermi_ev
fig, ax = plt.subplots(figsize=(6.0, 4.4))
ax.plot(bi2te3_axis, bi2te3_shift, color="tab:green", linewidth=0.4)
ax.axhline(0.0, color="0.4", linewidth=0.8)
ax.set_ylim(-4.0, 2.0)
ax.set_xlabel(r"K--$\Gamma$--K path distance ($\AA^{-1}$)")
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_title("Bi2Te3 bulk bands")
plt.show()
```



![png](91-anchor-band-structures_files/91-anchor-band-structures_21_0.png)




```python
bi4te3_axis = np.asarray(
    kpath_arc_length(make_kpath(bi4te3_bands.kpoints), bi4te3_geo)
)
bi4te3_shift = np.asarray(bi4te3_bands.eigenvalues) - bi4te3_fermi_ev
fig, ax = plt.subplots(figsize=(6.0, 4.4))
ax.plot(bi4te3_axis, bi4te3_shift, color="tab:brown", linewidth=0.4)
ax.axhline(0.0, color="0.4", linewidth=0.8)
ax.set_ylim(-4.0, 2.0)
ax.set_xlabel(r"M--$\Gamma$--M path distance ($\AA^{-1}$)")
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_title("Bi4Te3 bulk bands")
plt.show()
```



![png](91-anchor-band-structures_files/91-anchor-band-structures_22_0.png)




```python
bi4te3_slab_axis = np.asarray(
    kpath_arc_length(
        make_kpath(bi4te3_slab_bands.kpoints), bi4te3_slab_geo
    )
)
bi4te3_slab_shift = (
    np.asarray(bi4te3_slab_bands.eigenvalues) - bi4te3_slab_fermi_ev
)
fig, ax = plt.subplots(figsize=(6.0, 4.4))
ax.plot(
    bi4te3_slab_axis, bi4te3_slab_shift, color="tab:olive", linewidth=0.4
)
ax.axhline(0.0, color="0.4", linewidth=0.8)
ax.set_ylim(-4.0, 2.0)
ax.set_xlabel(r"M--$\Gamma$--M path distance ($\AA^{-1}$)")
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_title("Bi4Te3 five-layer slab bands")
plt.show()
```



![png](91-anchor-band-structures_files/91-anchor-band-structures_23_0.png)



## Compare the Densities of States

Total density of states for the Bi2Se3 bulk and slab calculations. The
slab curve adds the vacuum region.


```python
bi2se3_bulk_dos = read_doscar(
    str(DATA_ROOT / "Bi2Se3" / "Bulk" / "Output" / "DOSCAR")
)
fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.plot(
    np.asarray(bi2se3_bulk_dos.energy)
    - float(bi2se3_bulk_dos.fermi_energy),
    np.asarray(bi2se3_bulk_dos.total_dos),
    color="tab:blue",
)
ax.axvline(0.0, color="0.4", linewidth=0.8)
ax.set_xlim(-6.0, 4.0)
ax.set_xlabel(r"$E - E_F$ (eV)")
ax.set_ylabel("total DOS (states/eV)")
ax.set_title("Bi2Se3 bulk density of states")
plt.show()
```



![png](91-anchor-band-structures_files/91-anchor-band-structures_25_0.png)




```python
bi2se3_slab_dos = read_doscar(str(BI2SE3_DIR / "DOSCAR"))
fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.plot(
    np.asarray(bi2se3_slab_dos.energy)
    - float(bi2se3_slab_dos.fermi_energy),
    np.asarray(bi2se3_slab_dos.total_dos),
    color="tab:cyan",
)
ax.axvline(0.0, color="0.4", linewidth=0.8)
ax.set_xlim(-6.0, 4.0)
ax.set_xlabel(r"$E - E_F$ (eV)")
ax.set_ylabel("total DOS (states/eV)")
ax.set_title("Bi2Se3 slab density of states")
plt.show()
```



![png](91-anchor-band-structures_files/91-anchor-band-structures_26_0.png)



## Render ARPES-Style Intensity Maps

`dedupe_band_path` removes the repeated segment anchor that VASP line
mode writes at Gamma. `assemble_spectral_intensity_bands_chunk` folds
the absolute eigenvalues into occupied intensity. A 30 meV constant
self-energy sets the line shape, and the occupation uses 100 K. The
degenerate-value flag accepts the Kramers-degenerate slab eigenvalues
for this primal evaluation. `plot_arpes_with_kpath` draws each map with
the symmetry labels from the KPOINTS file.


```python
omega_ev = jnp.linspace(-1.4, 0.5, 261)
map_self_energy = make_self_energy_model(gamma=0.03)
bi2se3_map_bands, bi2se3_map_kinfo, _ = dedupe_band_path(
    bi2se3_bands, bi2se3_kinfo
)
bi2se3_map_dist = kpath_arc_length(
    make_kpath(bi2se3_map_bands.kpoints), bi2se3_geo
)
bi2se3_nk, bi2se3_nb = bi2se3_map_bands.eigenvalues.shape
bi2se3_weights = jnp.ones((bi2se3_nk, omega_ev.shape[0], bi2se3_nb))
bi2se3_intensity = assemble_spectral_intensity_bands_chunk(
    bi2se3_map_bands.eigenvalues,
    bi2se3_weights,
    omega_ev,
    map_self_energy,
    jnp.asarray(bi2se3_fermi_ev),
    100.0,
    allow_degenerate_value_only=True,
)
bi2se3_spectrum = make_arpes_spectrum(
    bi2se3_intensity,
    omega_ev,
    bi2se3_map_dist,
    kpoints_frac_to_cart(bi2se3_map_bands.kpoints, bi2se3_geo),
)
fig, ax = plt.subplots(figsize=(6.2, 4.8))
plot_arpes_with_kpath(
    bi2se3_spectrum,
    bi2se3_map_kinfo,
    ax=ax,
    cmap="magma",
    title="Bi2Se3 slab ARPES-style map",
)
plt.show()
```



![png](91-anchor-band-structures_files/91-anchor-band-structures_28_0.png)




```python
pdte2_map_bands, pdte2_map_kinfo, _ = dedupe_band_path(
    pdte2_mgm, pdte2_mgm_kinfo
)
pdte2_map_dist = kpath_arc_length(
    make_kpath(pdte2_map_bands.kpoints), pdte2_geo
)
pdte2_nk, pdte2_nb = pdte2_map_bands.eigenvalues.shape
pdte2_weights = jnp.ones((pdte2_nk, omega_ev.shape[0], pdte2_nb))
pdte2_intensity = assemble_spectral_intensity_bands_chunk(
    pdte2_map_bands.eigenvalues,
    pdte2_weights,
    omega_ev,
    map_self_energy,
    jnp.asarray(pdte2_fermi_ev),
    100.0,
    allow_degenerate_value_only=True,
)
pdte2_spectrum = make_arpes_spectrum(
    pdte2_intensity,
    omega_ev,
    pdte2_map_dist,
    kpoints_frac_to_cart(pdte2_map_bands.kpoints, pdte2_geo),
)
fig, ax = plt.subplots(figsize=(6.2, 4.8))
plot_arpes_with_kpath(
    pdte2_spectrum,
    pdte2_map_kinfo,
    ax=ax,
    cmap="magma",
    title="PdTe2 slab ARPES-style map",
)
plt.show()
```



![png](91-anchor-band-structures_files/91-anchor-band-structures_29_0.png)
